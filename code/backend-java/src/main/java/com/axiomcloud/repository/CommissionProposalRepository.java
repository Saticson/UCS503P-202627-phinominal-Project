package com.axiomcloud.repository;

import com.axiomcloud.model.CommissionProposalEntity;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface CommissionProposalRepository extends JpaRepository<CommissionProposalEntity, String> {

    List<CommissionProposalEntity> findByCommissionIdOrderByCreatedAtDesc(String commissionId);

    long countByCommissionId(String commissionId);
}
