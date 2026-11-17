package com.axiomcloud.service;

import com.axiomcloud.dto.MarketplaceListingDto;
import com.axiomcloud.model.MarketplaceListingEntity;
import com.axiomcloud.repository.MarketplaceListingRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.*;
import java.util.stream.Collectors;

@Service
public class MarketplaceService {

    private final MarketplaceListingRepository repository;

    public MarketplaceService(MarketplaceListingRepository repository) {
        this.repository = repository;
    }

    public List<MarketplaceListingDto> listListings(String modality, String query, Boolean verifiedOnly) {
        return repository.searchListings(modality, query, verifiedOnly)
                .stream()
                .map(MarketplaceListingEntity::toDto)
                .collect(Collectors.toList());
    }

    public Optional<MarketplaceListingDto> getListing(String id) {
        return repository.findById(id).map(MarketplaceListingEntity::toDto);
    }

    public List<MarketplaceListingDto> recommendListings(String modality, String taskType) {
        List<MarketplaceListingEntity> listings;
        if (modality != null && !modality.isBlank() && !modality.equalsIgnoreCase("ALL")) {
            listings = repository.findByModalityIgnoreCaseOrderByMetricScoreDesc(modality);
        } else {
            listings = repository.findAllByOrderByMetricScoreDesc();
        }

        return listings.stream()
                .filter(l -> taskType == null || taskType.isBlank() || taskType.equalsIgnoreCase(l.getTaskType()))
                .limit(4)
                .map(MarketplaceListingEntity::toDto)
                .collect(Collectors.toList());
    }

    @Transactional
    public MarketplaceListingDto publish(MarketplaceListingDto dto) {
        if (dto.getId() == null || dto.getId().isBlank()) {
            dto.setId("custom-" + UUID.randomUUID().toString().substring(0, 8));
        }
        if (dto.getSellerName() == null || dto.getSellerName().isBlank()) {
            dto.setSellerName("Community Seller");
        }
        if (dto.getPrice() == null || dto.getPrice().isBlank()) {
            dto.setPrice("Free");
        }
        if (dto.getDownloadUrl() == null || dto.getDownloadUrl().isBlank()) {
            dto.setDownloadUrl("/api/marketplace/" + dto.getId() + "/download");
        }
        MarketplaceListingEntity entity = MarketplaceListingEntity.fromDto(dto);
        MarketplaceListingEntity saved = repository.save(entity);
        return saved.toDto();
    }

    public int count() {
        return (int) repository.count();
    }
}
